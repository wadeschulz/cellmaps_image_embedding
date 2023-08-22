=====
Usage
=====

This page should provide information on how to use cellmaps_image_embedding

In a project
--------------

To use cellmaps_image_embedding in a project::

    import cellmaps_image_embedding
    

On the command line
---------------------

For information invoke :code:`cellmaps_image_embeddingcmd.py -h`

**Example usage**

The output directory for the image downloads is required (see `Cell Maps Image Downloader <https://github.com/idekerlab/cellmaps_imagedownloader/>`__). Optionally, a path to the image embedding model can be provided. 

.. code-block::
   # use wget to download model or directly visit url below to download the model file
   # to current directory
   wget https://github.com/CellProfiling/hpa_densenet/raw/main/models/bestfitting_default_model.pth
   
.. code-block::

   cellmaps_image_embeddingcmd.py ./cellmaps_image_embedding_outdir --inputdir ./cellmaps_imagedownloader_outdir 

Via Docker
---------------

**Example usage**


.. code-block::

   docker run -v `pwd`:`pwd` -w `pwd` idekerlab/cellmaps_image_embedding:0.1.0 cellmaps_image_embeddingcmd.py ./cellmaps_image_embedding_outdir --inputdir ./cellmaps_imagedownloader_outdir 


