from pathlib import Path

import fire

from cryo_pandda.align_model_to_map import fit_pdb_in_map


def _get_model_in_dir(dataset_dir):
    for file in dataset_dir.glob("*"):
        if file.suffix == "pdb":
            return file
    ...


def _get_mrc_in_dir(dataset_dir):
    for file in dataset_dir.glob("*"):
        if file.suffix == "mrc":
            return file


def __main__(data_dirs):
    data_dirs = Path(data_dirs).resolve()

    # Iterate over dataset directories, finding the mrc file and running alignment on it
    for dataset_dir in data_dirs.glob("*"):
        pdbFname = _get_model_in_dir(dataset_dir)
        mapFname = _get_mrc_in_dir(dataset_dir)
        pdbFnameOut = str(dataset_dir / "dimple.pdb")

        fit_pdb_in_map(
            pdbFname,
            mapFname,
            pdbFnameOut=pdbFnameOut,
            pdb2mapFname=None,
            resolution=3.,
            maskFname=None,
            n_iterations=100,
            verbose=False,
        )


if __name__ == "__main__":
    fire.Fire(__main__)