import multiprocessing
import os
import os.path as osp
import shutil
import sys
import tempfile
from typing import Optional

import numpy as np

from cryoUtils.atomicModel import MyStructure
from cryoUtils.cryoIO.images import data_and_md_from_mrcfile, data_to_mrcfile




def volumeCenterOfMassXYZ(vol, sampling_rate, xyz_ori_inA):
    """

    :param vol:
    :param sampling_rate:
    :param xyz_ori_inA: xyz origin of coordinates
    :return: xyz coords in A for the volume centre of mass
    """
    vox_idxs = np.meshgrid(*[np.arange(float(x)) for x in vol.shape], indexing="ij")
    vox_coords = np.stack(vox_idxs, -1) * sampling_rate
    vox_coords -= np.flip(xyz_ori_inA) # voxels are zyx, but origin of coordinates is xyz, so move coord_ori_inA -> zyx

    #rescale vol so that it stats at 0
    vol -= vol.min()
    weighted_coords = vol.reshape(vol.shape+(1,)) * vox_coords
    center_of_mass = weighted_coords.sum((0,1,2))/vol.sum()
    center_of_mass = np.flip(center_of_mass)
    return center_of_mass

def fit_pdb_in_map(pdbFname:str, mapFname:str, pdbFnameOut:str=None, pdb2mapFname:str=None,
                   resolution:float=3.,
                   maskFname:Optional[str]=None, n_iterations:int=100, verbose:bool=False):
    """

    :param pdbFname: the pdb to align against the reference mapFname
    :param mapFname: the reference mapFname
    :param pdbFnameOut: If None, no output pdb is written
    :param resolution: to convert pdbFname to a map
    :param maskFname:
    :param n_iterations:
    :param verbose:
    :return:
    """
    import gemmi

    #Get the center of mass of the pdb and the volume
    pdbObj = MyStructure(pdbFname)
    pdb_center = pdbObj.centerOfMass()  # In A
    vol, md = data_and_md_from_mrcfile(mapFname)
    vol_center = volumeCenterOfMassXYZ(vol, sampling_rate=md["sampling_rate"], xyz_ori_inA=md["xyz_ori_inA"])
    center_translation = vol_center - pdb_center
    if verbose:
        print(f"vol_center {vol_center}")
        print(f"pdb_center {pdb_center}")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdbObj.transform(np.eye(3), center_translation)
        newPdbFname = osp.join(tmpdir, osp.basename(pdbFname))
        pdbObj.save(newPdbFname)
        # print(newPdbFname)
        queue = multiprocessing.Queue()
        def workerFun():
            from emda.emda_methods import overlay_maps  # Using my fork https://gitlab.com/rsanchezgarcia/emda to correct a bug
            os.chdir(tmpdir)
            if not verbose:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
            out = overlay_maps(
                maplist=[mapFname, newPdbFname],
                modelres=resolution,
                masklist=None if maskFname is None else [maskFname],
                ncy=n_iterations, fitres=2., nocom=False
            )
            queue.put(out[1:])
            print("Alignment completed")
        p = multiprocessing.Process(target=workerFun)
        p.start()
        p.join()
        try:
            from queue import Empty
            try:
                rot_mat, translation = queue.get(timeout=1)
            except Empty:
                raise RuntimeWarning("fit_pdb_in_map failed")
            # print("rot_mat, translation,", rot_mat, translation)
        except TimeoutError:
            raise RuntimeWarning("fit_pdb_in_map failed")
        finally:
            queue.close()
        structure = gemmi.read_structure(osp.join(tmpdir, "emda_transformed_model_0.cif")) #TODO: keep the map if asked
        # print("structure", structure)

        if pdbFnameOut:
            newPdbFname = pdbFnameOut
        else:
            newPdbFname = osp.join(tmpdir, f"emda_transformed_{osp.basename(pdbFname)}_0.pdb")

        if pdb2mapFname:
            shutil.copyfile(osp.join(tmpdir, "fitted_map_1.mrc"), pdb2mapFname)
        structure.write_minimal_pdb(newPdbFname)
        pdbObj = MyStructure(newPdbFname)
    return pdbObj

def fit_map_in_map(queryMapFname:str, refMapFname:str, mapFnameOut:str, maskFname:Optional[str]=None,
                   n_iterations:int=100, verbose:bool=False): #TODO: refactor, as both fit_map_in_map and fit_pdb_in_map are almost identical
    """

    :param queryMapFname: the pdb to align against the reference mapFname
    :param refMapFname: the reference mapFname
    :param mapFnameOut: the output name of the aligned map
    :param maskFname:
    :param n_iterations:
    :param verbose:
    :return:
    """
    import gemmi

    #Get the center of mass of the pdb and the volume


    vol_query, md_query = data_and_md_from_mrcfile(queryMapFname)
    vol_ref, md_ref = data_and_md_from_mrcfile(refMapFname)

    query_center = volumeCenterOfMassXYZ(vol_query, sampling_rate=md_query["sampling_rate"],
                                         xyz_ori_inA=md_query["xyz_ori_inA"])

    ref_center = volumeCenterOfMassXYZ(vol_ref, sampling_rate=md_ref["sampling_rate"],
                                         xyz_ori_inA=md_ref["xyz_ori_inA"])

    center_translation = ref_center - query_center
    if verbose:
        print(f"ref_center {ref_center}")
        print(f"query_center {query_center}")

    with tempfile.TemporaryDirectory() as tmpdir:
        newQueryFname = osp.join(tmpdir, osp.basename(queryMapFname))
        data_to_mrcfile(newQueryFname, vol_query, sampling_rate=md_query["sampling_rate"],
                        xyz_ori_inA=md_query["xyz_ori_inA"] - center_translation)
        queue = multiprocessing.Queue()
        def workerFun():
            from emda.emda_methods import overlay_maps  # Using my fork https://gitlab.com/rsanchezgarcia/emda to correct a bug
            os.chdir(tmpdir)
            if not verbose:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
            out = overlay_maps(
                maplist=[refMapFname, newQueryFname],
                masklist=None if maskFname is None else [maskFname],
                ncy=n_iterations, fitres=2., nocom=False
            )
            queue.put(out[1:])
            print("Alignment completed")

        p = multiprocessing.Process(target=workerFun, )
        p.start()
        p.join()
        try:
            from queue import Empty
            try:
                rot_mat, translation = queue.get(timeout=1)
            except Empty:
                raise RuntimeWarning("fit_pdb_in_map failed")
            # print("rot_mat, translation,", rot_mat, translation)
        except TimeoutError:
            raise RuntimeWarning("fit_pdb_in_map failed")
        queue.close()
        aligned_vol, aligned_md = data_and_md_from_mrcfile(osp.join(tmpdir, f"fitted_map_1.mrc"))
        if mapFnameOut:
            data_to_mrcfile(mapFnameOut, aligned_vol, sampling_rate=aligned_md["sampling_rate"],
                            xyz_ori_inA=aligned_md["xyz_ori_inA"])
    return aligned_vol, aligned_md

def _test():
    refmap = "/home/sanchezg/cryo/myProjects/micSimulations/data/temSimulatorOutput/example1/simulatedMaps/dpp11-x0032_0A_bound_corrected.mrc"
    pdbFname = "/home/sanchezg/cryo/myProjects/micSimulations/data/pdbExamples/processed/dpp11-x0032_0A_bound.pdb"
    pdbFnameOut = "/tmp/kk.pdb"
    fit_pdb_in_map(pdbFname, mapFname=refmap, pdbFnameOut=pdbFnameOut, resolution=3., maskFname=None, n_iterations=100)

def fit_pdb_in_map_test(pdb, dmap, pdb_out):
    fit_pdb_in_map(str(Path(pdb).resolve()),
                   mapFname=str(Path(dmap).resolve()), pdbFnameOut=str(Path(pdb_out).resolve()), resolution=3., maskFname=None, n_iterations=100, verbose=True)
import fire
if __name__ == "__main__":
    fire.Fire(fit_pdb_in_map_test)

    # from argParseFromDoc import AutoArgumentParser, get_parser_from_function
    # parser = AutoArgumentParser("alignMaps")
    #
    # subparsers = parser.add_subparsers(help='command: fit_pdb_in_map or fit_map_in_map', required=True, dest='command')
    # pdb_in_map_parser = subparsers.add_parser('fit_pdb_in_map', help='fit a pdb in a map. Rigid body only.')
    # get_parser_from_function(fit_pdb_in_map, parser=pdb_in_map_parser)
    #
    # map_in_map_parser = subparsers.add_parser('fit_map_in_map', help='fit a map in a reference map. Rigid body only.')
    # get_parser_from_function(fit_map_in_map, parser=map_in_map_parser)
    #
    # arguments = parser.parse_args()
    # if arguments.command == "fit_pdb_in_map":
    #     del arguments.command
    #     fit_pdb_in_map(**vars(arguments))
    # elif arguments.command == "fit_map_in_map":
    #     del arguments.command
    #     fit_map_in_map(**vars(arguments))
    # else:
    #     raise ValueError(f"Command not valid {arguments.command}")