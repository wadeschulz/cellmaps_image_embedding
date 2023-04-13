#! /usr/bin/env python

import os
import csv
import time
import logging
import random
import cellmaps_image_embedding
from cellmaps_utils import cellmaps_io
from cellmaps_image_embedding.exceptions import CellMapsImageEmbeddingError

logger = logging.getLogger(__name__)


class CellmapsImageEmbeddingRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None,
                 imagedir=None,
                 image_gene_node_attributes=None):
        """
        Constructor

        :param exitcode: value to return via :py:meth:`.CellmapsImageEmbeddingRunner.run` method
        :type int:
        """
        logger.debug('In constructor')
        self._outdir = outdir
        self._imagedir = imagedir
        self._image_gene_node_attributes = image_gene_node_attributes
        self._start_time = int(time.time())
        self._dimensions = 1024

    def run(self):
        """
        Runs cellmaps_image_embedding


        :return:
        """
        logger.debug('In run method')
        if self._outdir is None:
            raise CellMapsImageEmbeddingError('outdir must be set')

        if not os.path.isdir(self._outdir):
            os.makedirs(self._outdir, mode=0o755)

        cellmaps_io.setup_filelogger(outdir=self._outdir,
                                     handlerprefix='cellmaps_image_embedding')
        cellmaps_io.write_task_start_json(outdir=self._outdir,
                                          start_time=self._start_time,
                                          data={'image_gene_node_attributes': str(self._image_gene_node_attributes),
                                                'imagedir': self._imagedir},
                                          version=cellmaps_image_embedding.__version__)
        exit_status = 99
        try:
            if self._imagedir is None:
                raise CellMapsImageEmbeddingError('imagedir must be set')

            if self._image_gene_node_attributes is None:
                raise CellMapsImageEmbeddingError('image_gene_node_attributes must be set')

            uniq_genes = set()
            with open(self._image_gene_node_attributes, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    uniq_genes.add(row[0])

            embedding_file = os.path.join(self._outdir, 'image_emd.tsv')
            with open(embedding_file, 'w') as f:
                headerline = ['']
                for x in range(1, 1025):
                    headerline.append(str(x))
                f.write('\t'.join(headerline) + '\n')
                for gene in uniq_genes:
                    embedding = [gene]
                    for cntr in range(self._dimensions):
                        embedding.append(str(random.random()))
                    f.write('\t'.join(embedding) + '\n')
            exit_status = 0
        finally:
            cellmaps_io.write_task_finish_json(outdir=self._outdir,
                                               start_time=self._start_time,
                                               status=exit_status)

        return exit_status

