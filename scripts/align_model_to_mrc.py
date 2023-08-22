import os
from pathlib import Path

import fire

from cryo_pandda.align_model_to_map import fit_pdb_in_map


def _get_model_in_dir(dataset_dir):
    for file in dataset_dir.glob("*"):
        if file.suffix == ".pdb":
            return file
    ...


def _get_mrc_in_dir(dataset_dir):
    for file in dataset_dir.glob("*"):
        if file.suffix == ".mrc":
            return file


def __main__(data_dirs):
    data_dirs = Path(data_dirs).resolve()

    # Iterate over dataset directories, finding the mrc file and running alignment on it
    for dataset_dir in data_dirs.glob("*"):
        pdbFname = str(_get_model_in_dir(dataset_dir))
        mapFname = str(_get_mrc_in_dir(dataset_dir))
        pdbFnameOut = str(dataset_dir / "dimple.pdb")
        if Path(pdbFnameOut).exists():
            print(f"Already have aligned pdb! Skipping!")
            continue

        print(f"Directory name: {dataset_dir}")
        print(f"PDB file name: {pdbFname}")
        print(f"Map file name: {mapFname}")
        print(f"Output name: {pdbFnameOut}")

        fit_pdb_in_map(
            pdbFname,
            mapFname,
            pdbFnameOut=pdbFnameOut,
            pdb2mapFname=None,
            resolution=3.,
            maskFname=None,
            n_iterations=100,
            verbose=True,
        )


    print(f"Finished!")

if __name__ == "__main__":
    fire.Fire(__main__)
