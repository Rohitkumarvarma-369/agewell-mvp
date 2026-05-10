"""Offline BrainIAC preprocessing and cache utilities."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import SimpleITK as sitk

from agewell._common.paths import models_root, repo_root
from agewell.services._common.cache import derivative_root, imaging_cache_stem

ADNI_PASSTHROUGH_VERSION = "preprocessed_v1_passthrough"
IXI_HDBET_VERSION = "hdbet_v1_ixi_mni"
FULL_PREPROCESS_VERSION = "hdbet_v1_n4_nihpd"


@dataclass(frozen=True)
class PreprocessPlan:
    """Resolved preprocessing branch for a scan."""

    version: str
    run_n4: bool
    run_registration: bool
    run_hdbet: bool
    passthrough: bool = False


@dataclass(frozen=True)
class PreprocessOutput:
    """Filesystem outputs from preprocessing a scan."""

    preprocessed_uri: str
    preprocess_version: str
    registered_uri: str | None = None
    mask_uri: str | None = None
    brain_volume_ml: float | None = None
    registration_mi: float | None = None
    cache_hit: bool = False
    qc_reasons: tuple[str, ...] = ()


def plan_for_cohort(cohort: str) -> PreprocessPlan:
    """Return the preprocessing plan for a cohort."""
    if cohort == "ADNI_NIFTI":
        return PreprocessPlan(
            version=ADNI_PASSTHROUGH_VERSION,
            run_n4=False,
            run_registration=False,
            run_hdbet=False,
            passthrough=True,
        )
    if cohort == "IXI":
        return PreprocessPlan(
            version=IXI_HDBET_VERSION,
            run_n4=False,
            run_registration=False,
            run_hdbet=True,
        )
    return PreprocessPlan(
        version=FULL_PREPROCESS_VERSION,
        run_n4=True,
        run_registration=True,
        run_hdbet=True,
    )


def preprocess_scan(
    *,
    cohort: str,
    nifti_uri: str,
    root: Path | None = None,
    atlas_path: Path | None = None,
) -> PreprocessOutput:
    """Run the cohort-specific BrainIAC preprocessing branch."""
    plan = plan_for_cohort(cohort)
    if plan.passthrough:
        return PreprocessOutput(
            preprocessed_uri=nifti_uri,
            preprocess_version=plan.version,
            cache_hit=True,
        )

    cache_root = derivative_root("brainiac_preprocess", root=root) / plan.version
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = imaging_cache_stem(nifti_uri, plan.version)
    preprocessed_path = cache_root / f"{stem}.nii.gz"
    registered_path = cache_root / f"{stem}_registered.nii.gz"
    mask_path = cache_root / f"{stem}_mask.nii.gz"

    if preprocessed_path.exists():
        return PreprocessOutput(
            preprocessed_uri=str(preprocessed_path),
            preprocess_version=plan.version,
            registered_uri=str(registered_path) if registered_path.exists() else None,
            mask_uri=str(mask_path) if mask_path.exists() else None,
            brain_volume_ml=_brain_volume_ml(mask_path) if mask_path.exists() else None,
            cache_hit=True,
        )

    source = Path(nifti_uri)
    with tempfile.TemporaryDirectory(prefix="agewell-brainiac-pre-") as tmp:
        tmp_path = Path(tmp)
        hdbet_input = tmp_path / "hdbet_input"
        hdbet_output = tmp_path / "hdbet_output"
        hdbet_input.mkdir()
        hdbet_output.mkdir()
        hdbet_ready = hdbet_input / f"{stem}_0000.nii.gz"
        registration_mi = None

        if plan.run_registration:
            atlas = atlas_path or default_atlas_path()
            registration_mi = register_to_atlas(
                source,
                hdbet_ready,
                atlas,
                run_n4=plan.run_n4,
            )
            shutil.copy2(hdbet_ready, registered_path)
        else:
            _write_nifti_gz(source, hdbet_ready)

        run_hdbet(hdbet_input, hdbet_output)
        hdbet_image = hdbet_output / hdbet_ready.name
        hdbet_mask = hdbet_output / hdbet_ready.name.replace(".nii.gz", "_mask.nii.gz")
        shutil.copy2(hdbet_image, preprocessed_path)
        if hdbet_mask.exists():
            shutil.copy2(hdbet_mask, mask_path)

    return PreprocessOutput(
        preprocessed_uri=str(preprocessed_path),
        preprocess_version=plan.version,
        registered_uri=str(registered_path) if registered_path.exists() else None,
        mask_uri=str(mask_path) if mask_path.exists() else None,
        brain_volume_ml=_brain_volume_ml(mask_path) if mask_path.exists() else None,
        registration_mi=registration_mi,
        cache_hit=False,
    )


def default_atlas_path() -> Path:
    """Return the DVC-tracked BrainIAC NIHPD T1 atlas path."""
    return models_root() / "brainiac" / "atlases" / "nihpd_asym_13.0-18.5_t1w.nii"


def register_to_atlas(
    moving_path: Path,
    output_path: Path,
    atlas_path: Path,
    *,
    run_n4: bool,
) -> float:
    """Rigid-register one scan to the BrainIAC NIHPD atlas."""
    fixed_img = sitk.ReadImage(str(atlas_path), sitk.sitkFloat32)
    moving_img = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    if run_n4:
        moving_img = sitk.N4BiasFieldCorrection(moving_img)

    fixed_img = _resample_1mm(fixed_img)
    transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)

    final_transform = registration_method.Execute(fixed_img, moving_img)
    moving_img_resampled = sitk.Resample(
        moving_img,
        fixed_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID(),
    )
    sitk.WriteImage(moving_img_resampled, str(output_path))
    return float(registration_method.GetMetricValue())


def run_hdbet(input_dir: Path, output_dir: Path) -> None:
    """Run vendored HD-BET in BrainIAC-compatible fast mode."""
    _ensure_vendor_on_path()
    os.environ.setdefault("HDBET_PARAMS_DIR", str(models_root() / "brainiac" / "hdbet"))
    from HD_BET.hd_bet import hd_bet  # type: ignore[import-not-found]

    device = "0" if _cuda_available() else "cpu"
    hd_bet(str(input_dir), str(output_dir), mode="fast", device=device, tta=0, save_mask=1)


def _write_nifti_gz(source: Path, dest: Path) -> None:
    image = sitk.ReadImage(str(source))
    sitk.WriteImage(image, str(dest))


def _ensure_vendor_on_path() -> None:
    vendor_root = repo_root() / "src" / "agewell" / "vendor"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resample_1mm(image: sitk.Image) -> sitk.Image:
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_spacing = (1, 1, 1)
    new_size = [round((old_size[i] * old_spacing[i]) / float(new_spacing[i])) for i in range(3)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetOutputPixelType(sitk.sitkFloat32)
    return resample.Execute(image)


def _brain_volume_ml(mask_path: Path) -> float:
    mask = cast(Any, nib.load(str(mask_path)))
    data = mask.get_fdata() > 0
    voxel_volume_ml = (
        float(
            abs(
                mask.header.get_zooms()[0] * mask.header.get_zooms()[1] * mask.header.get_zooms()[2]
            )
        )
        / 1000
    )
    return float(data.sum() * voxel_volume_ml)
