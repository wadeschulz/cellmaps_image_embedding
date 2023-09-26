=====
Usage
=====

Generates image embeddings of immunofluorescent labeled images from the Human Protein Atlas that were downloaded by the cellmaps_imagedownloader package.

In a project
--------------

To use cellmaps_image_embedding in a project::

    import cellmaps_image_embedding


On the command line
---------------------

For information invoke :code:`cellmaps_image_embeddingcmd.py -h`

**Usage**

.. code-block::

  cellmaps_image_embeddingcmd.py [outdir] [--inputdir IMAGEDOWNLOADER_OUT_DIR] [OPTIONS]

**Arguments**

- ``outdir``
    The directory where the output will be written to.

*Required*

- ``--inputdir``
    Directory with blue, red, yellow, and green image directories (output of cellmaps_image_downloader package).

*Optional*

- ``--model_path``
    URL or path to model file for image embedding. Default is a link to a specific Densenet model.

- ``--name``
    Name of this run, needed for FAIRSCAPE. If unset, name value from the directory specified by --inputdir will be used.

- ``--organization_name``
    Name of the organization running this tool, needed for FAIRSCAPE. If unset, the organization name specified in --inputdir directory will be used.

- ``--project_name``
    Name of the project running this tool, needed for FAIRSCAPE. If unset, the project name specified in --input directory will be used.

- ``--fold``
    Image node attribute file fold to use. Default is 1.

- ``--fake_embedder``
    If set, generate fake embedding.

- ``--dimensions``
    Dimensions of generated embedding vector. Default is 1024.

- ``--suffix``
    Suffix for image files. Default is .jpg.

- ``--logconf``
    Path to the Python logging configuration file in the specified format. Setting this overrides the -v parameter which uses the default logger.

- ``--verbose``, ``-vv``
    Increases verbosity of logger to standard error for log messages in this module. Logging levels: -v = ERROR, -vv = WARNING, -vvv = INFO, -vvvv = DEBUG, -vvvvv = NOTSET. Default is no logging.

- ``--version``
    Display the version of the package.

**Example usage**

The output directory for the image downloads is required (see `Cell Maps Image Downloader <https://github.com/idekerlab/cellmaps_imagedownloader/>`__). Optionally, a path to the image embedding model can be provided.

.. code-block::
   # use wget to download model or directly visit url below to download the model file
   # to current directory
   wget https://github.com/CellProfiling/hpa_densenet/raw/main/models/bestfitting_default_model.pth

.. code-block::

   cellmaps_image_embeddingcmd.py ./cellmaps_image_embedding_outdir --inputdir ./cellmaps_imagedownloader_outdir --fold 1

Via Docker
---------------

**Example usage**


.. code-block::

   docker run -v `pwd`:`pwd` -w `pwd` idekerlab/cellmaps_image_embedding:0.1.0 cellmaps_image_embeddingcmd.py ./cellmaps_image_embedding_outdir --inputdir ./cellmaps_imagedownloader_outdir


