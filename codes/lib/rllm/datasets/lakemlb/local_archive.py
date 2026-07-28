from pathlib import Path
from typing import Iterable
from zipfile import ZipFile


def extract_local_archive(
    dataset_name: str,
    relationship: str,
    raw_dir: str,
    expected_filenames: Iterable[str],
) -> None:
    """Extract a repository-local, flat dataset archive into ``raw_dir``."""
    repository_root = Path(__file__).resolve().parents[5]
    archive_path = (
        repository_root
        / "benckmark"
        / f"{relationship}_based"
        / f"{dataset_name}.zip"
    )
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Local dataset archive not found: {archive_path}"
        )

    expected = set(expected_filenames)
    with ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        member_names = [member.filename for member in members]

        if any(Path(name).name != name for name in member_names):
            raise RuntimeError(
                f"Dataset archive must be flat: {archive_path}"
            )
        if len(member_names) != len(set(member_names)):
            raise RuntimeError(
                f"Dataset archive contains duplicate filenames: {archive_path}"
            )
        if set(member_names) != expected:
            missing = sorted(expected - set(member_names))
            unexpected = sorted(set(member_names) - expected)
            raise RuntimeError(
                f"Invalid contents in {archive_path}; "
                f"missing={missing}, unexpected={unexpected}"
            )

        destination = Path(raw_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            archive.extract(member, destination)
