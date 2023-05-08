#! /usr/bin/env python

import argparse
import sys
import json
import logging
import logging.config
from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_image_embedding
from cellmaps_image_embedding.runner import CellmapsImageEmbeddingRunner
from cellmaps_image_embedding.runner import FakeEmbeddingGenerator

logger = logging.getLogger(__name__)


def _parse_arguments(desc, args):
    """
    Parses command line arguments

    :param desc: description to display on command line
    :type desc: str
    :param args: command line arguments usually :py:func:`sys.argv[1:]`
    :type args: list
    :return: arguments parsed by :py:mod:`argparse`
    :rtype: :py:class:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=constants.ArgParseFormatter)
    parser.add_argument('outdir', help='Output directory')
    parser.add_argument('--inputdir', required=True,
                        help='Directory where blue, red, yellow, and '
                             'green image directories reside ')
    parser.add_argument('--model_path', type=str, default='/opt/densenet/models/model.pth',
                        help='Path to model file to use (set to one in container)')
    parser.add_argument('--name',
                        help='Name of this run, needed for FAIRSCAPE. If '
                             'unset, name value from specified '
                             'by --inputdir directory will be used')
    parser.add_argument('--organization_name',
                        help='Name of organization running this tool, needed '
                             'for FAIRSCAPE. If unset, organization name specified '
                             'in --inputdir directory will be used')
    parser.add_argument('--project_name',
                        help='Name of project running this tool, needed for '
                             'FAIRSCAPE. If unset, project name specified '
                             'in --input directory will be used')

    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = ERROR, -vv = WARNING, -vvv = INFO, '
                             '-vvvv = DEBUG, -vvvvv = NOTSET (default no '
                             'logging)')
    parser.add_argument('--version', action='version',
                        version=('%(prog)s ' +
                                 cellmaps_image_embedding.__version__))

    return parser.parse_args(args)


def main(args):
    """
    Main entry point for program

    :param args: arguments passed to command line usually :py:func:`sys.argv[1:]`
    :type args: list

    :return: return value of :py:meth:`cellmaps_image_embedding.runner.CellmapsImageEmbeddingRunner.run`
             or ``2`` if an exception is raised
    :rtype: int
    """
    desc = """
Version {version}

Invokes run() method on CellmapsImageEmbeddingRunner


    """.format(version=cellmaps_image_embedding.__version__)
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]
    theargs.version = cellmaps_image_embedding.__version__

    try:
        logutils.setup_cmd_logging(theargs)
        gen = FakeEmbeddingGenerator(theargs.inputdir, dimensions=1024)
        return CellmapsImageEmbeddingRunner(outdir=theargs.outdir,
                                            inputdir=theargs.inputdir,
                                            embedding_generator=gen,
                                            name=theargs.name,
                                            project_name=theargs.project_name,
                                            organization_name=theargs.organization_name).run()
    except Exception as e:
        logger.exception('Caught exception: ' + str(e))
        return 2
    finally:
        logging.shutdown()


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
