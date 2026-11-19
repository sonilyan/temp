"""
Physical deletion for ComfyUI's App Mode > Assets > Generated items.

Install:
    Copy this file directly into:
        ComfyUI/custom_nodes/asset_delete_physical.py

Then restart ComfyUI.

This hooks the existing DELETE /api/assets/{id} handler's service function.
After ComfyUI successfully deletes an asset record, it removes the underlying
file only when:
  - the file is inside ComfyUI's configured OUTPUT directory; and
  - no remaining asset record references the same content row.

It deliberately does NOT delete files from input, temp, models, or arbitrary
paths. If the installed ComfyUI asset API differs, the patch logs a warning
and leaves the normal delete behavior untouched.
"""

import logging
import os
from pathlib import Path

_LOG = logging.getLogger("asset_delete_physical")
_PATCH_MARKER = "_asset_delete_physical_patched"


def _is_inside(path: str, root: str) -> bool:
    """True only when path resolves inside root (not root itself)."""
    try:
        p = os.path.normcase(os.path.realpath(path))
        r = os.path.normcase(os.path.realpath(root))
        return p != r and os.path.commonpath((p, r)) == r
    except (OSError, ValueError, TypeError):
        return False


def _install_patch():
    try:
        import folder_paths
        from sqlalchemy import func, select

        from app.assets.api import routes
        from app.assets.database.models import Asset
    except Exception:
        _LOG.exception("Could not import ComfyUI asset API; physical-delete patch not installed.")
        return

    if getattr(routes.delete_asset_reference, _PATCH_MARKER, False):
        return

    original_delete = routes.delete_asset_reference

    def delete_with_physical_file(reference_id, *args, **kwargs):
        # Capture the content row/path before the original operation removes
        # the asset record.
        content_id = None
        file_path = None

        try:
            with routes.create_session() as session:
                record = session.get(Asset, str(reference_id))
                if record is not None:
                    content_id = record.content_id
                    file_path = record.content.path if record.content is not None else None
        except Exception:
            _LOG.exception(
                "Could not inspect asset %s before deletion; using normal record deletion only.",
                reference_id,
            )

        # Preserve ComfyUI's own validation, database behavior, and return value.
        deleted = original_delete(reference_id, *args, **kwargs)
        if not deleted:
            return deleted

        if not content_id or not file_path:
            _LOG.info("Asset %s deleted; no local file path was available.", reference_id)
            return deleted

        output_root = folder_paths.get_output_directory()
        if not output_root or not _is_inside(file_path, output_root):
            _LOG.info(
                "Asset %s deleted; keeping file outside output directory: %s",
                reference_id,
                file_path,
            )
            return deleted

        # Content can be shared by multiple asset records. Only unlink it when
        # no other record still points at the same content row.
        try:
            with routes.create_session() as session:
                remaining = session.scalar(
                    select(func.count())
                    .select_from(Asset)
                    .where(Asset.content_id == content_id)
                )
        except Exception:
            _LOG.exception(
                "Asset %s record deleted, but could not verify remaining references; "
                "keeping file to avoid accidental data loss: %s",
                reference_id,
                file_path,
            )
            return deleted

        if remaining:
            _LOG.info(
                "Asset %s deleted; keeping shared file (%s remaining reference(s)): %s",
                reference_id,
                remaining,
                file_path,
            )
            return deleted

        try:
            # Re-check immediately before unlinking, including symlink resolution.
            if not _is_inside(file_path, output_root):
                _LOG.warning("Refusing to delete path that is outside output: %s", file_path)
                return deleted

            path = Path(file_path)
            if path.is_file():
                path.unlink()
                _LOG.info("Deleted generated asset file: %s", file_path)
            else:
                _LOG.info("Asset file is already absent: %s", file_path)
        except Exception:
            # The API record deletion already succeeded. Do not turn that into
            # an HTTP failure; leave a clear diagnostic in the server log.
            _LOG.exception("Asset record %s deleted, but file removal failed: %s", reference_id, file_path)

        return deleted

    setattr(delete_with_physical_file, _PATCH_MARKER, True)
    routes.delete_asset_reference = delete_with_physical_file

    _LOG.warning(
        "Physical asset deletion patch installed: App Mode asset deletes will "
        "also remove unshared files under the output directory."
    )


_install_patch()

# ComfyUI's custom-node loader expects these names on imported modules.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
