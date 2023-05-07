#! /usr/bin/env python

import os
import time
import logging
import subprocess
import cellmaps_image_embedding
from cellmaps_utils import logutils
from cellmaps_utils.provenance import ProvenanceUtil
from cellmaps_image_embedding.exceptions import CellMapsImageEmbeddingError

logger = logging.getLogger(__name__)


class CellmapsImageEmbeddingRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None,
                 inputdir=None,
                 pythonbinary='/opt/conda/bin/python',
                 predict='/opt/densenet/predict/predict_d121.py',
                 model_path='/opt/densenet/models/model.pth',
                 suffix='jpg',
                 name=cellmaps_image_embedding.__name__,
                 organization_name=None,
                 project_name=None,
                 provenance_utils=ProvenanceUtil()):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsImageEmbeddingRunner.run` method
        :type int:
        """
        logger.debug('In constructor')
        self._outdir = outdir
        self._inputdir = inputdir
        self._start_time = int(time.time())
        self._dimensions = 1024
        self._pythonbinary = pythonbinary
        self._predict = predict
        self._model_path = model_path
        self._suffix = suffix
        self._name = name
        self._project_name = project_name
        self._organization_name = organization_name
        self._provenance_utils = provenance_utils
        self._softwareid = None

    def _run_cmd(self, cmd):
        """
        Runs hidef command as a command line process
        :param cmd_to_run: command to run as list
        :type cmd_to_run: list
        :return: (return code, standard out, standard error)
        :rtype: tuple
        """
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        out, err = p.communicate()

        return p.returncode, out, err

    def _get_image_id_list(self):
        """
        Looks at red directory under image directory to
        get a list of image ids which are the file names
        in that directory with last ``_`` and everything to
        the right of it removed from the file name
        :return:
        """
        image_set = set()
        red_image_dir = os.path.join(self._inputdir, 'red')
        for entry in os.listdir(red_image_dir):
            if not entry.endswith(self._suffix):
                continue
            if not os.path.isfile(os.path.join(red_image_dir, entry)):
                continue
            image_set.add(entry[: entry.rfind('_')])
        return list(image_set)

    def _create_run_crate(self):
        """
        Creates rocrate for output directory

        :raises CellMapsProvenanceError: If there is an error
        """
        # TODO: If organization or project name is unset need to pull from input rocrate
        org_name = self._organization_name
        if org_name is None:
            org_name = 'TODO BETTER SET THIS via input rocrate'

        proj_name = self._project_name
        if proj_name is None:
            proj_name = 'TODO BETTER SET THIS via input rocrate'
        try:
            self._provenance_utils.register_rocrate(self._outdir,
                                                    name='cellmaps_image_embedding',
                                                    organization_name=org_name,
                                                    project_name=proj_name)
        except TypeError as te:
            raise CellMapsImageEmbeddingError('Invalid provenance: ' + str(te))
        except KeyError as ke:
            raise CellMapsImageEmbeddingError('Key missing in provenance: ' + str(ke))

    def _register_software(self):
        """
        Registers this tool

        :raises CellMapsImageEmbeddingError: If fairscape call fails
        """
        self._softwareid = self._provenance_utils.register_software(self._outdir,
                                                                    name=self._name,
                                                                    description=cellmaps_image_embedding.__description__,
                                                                    author=cellmaps_image_embedding.__author__,
                                                                    version=cellmaps_image_embedding.__version__,
                                                                    file_format='.py',
                                                                    url=cellmaps_image_embedding.__repo_url__)

    def _register_computation(self):
        """
        # Todo: added inused dataset, software and what is being generated
        :return:
        """
        self._provenance_utils.register_computation(self._outdir,
                                                    name=cellmaps_image_embedding.__name__ + ' computation',
                                                    run_by=str(os.getlogin()),
                                                    command=str(self._input_data_dict),
                                                    description='run of ' + cellmaps_image_embedding.__name__,
                                                    used_software=[self._softwareid])
                                                    #used_dataset=[self._unique_datasetid, self._samples_datasetid],
                                                    #generated=[self._image_gene_attrid])

    def run(self):
        """
        Runs cellmaps_image_embedding


        :return:
        """
        exitcode = 99
        try:
            logger.debug('In run method')
            if self._outdir is None:
                raise CellMapsImageEmbeddingError('outdir must be set')

            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir, mode=0o755)

            logutils.setup_filelogger(outdir=self._outdir,
                                      handlerprefix='cellmaps_image_embedding')
            logutils.write_task_start_json(outdir=self._outdir,
                                           start_time=self._start_time,
                                           data={'imagedir': self._inputdir},
                                           version=cellmaps_image_embedding.__version__)

            if self._inputdir is None:
                raise CellMapsImageEmbeddingError('imagedir must be set')

            self._create_run_crate()

            # Todo: uncomment when fixed
            # register software fails due to this bug:
            # https://github.com/fairscape/fairscape-cli/issues/7
            # self._register_software()

            # just run a single command for now
            logger.info('Running command: ')
            cmd = [self._pythonbinary, self._predict,
                   '--image_dir', self._inputdir,
                   '--out_dir', self._outdir]
            exit_status, out, err = self._run_cmd(cmd=cmd)
            if out is not None:
                logger.debug(str(out))
            if err is not None:
                logger.error(str(err))

            # Todo: uncomment when above work
            # Above registrations need to work for this to work
            # register computation
            # self._register_computation()
        finally:
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            status=exitcode)

        return exitcode

