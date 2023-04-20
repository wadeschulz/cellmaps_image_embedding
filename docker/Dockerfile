# Use an official Python runtime as a parent image
FROM continuumio/anaconda3

RUN apt-get --allow-releaseinfo-change update
RUN apt-get install -y build-essential python3-opencv

RUN pip install https://github.com/idekerlab/cellmaps_utils/archive/master.zip

RUN mkdir /tmp/cellmaps_image_embedding
COPY ./ /tmp/cellmaps_image_embedding/
RUN pip install /tmp/cellmaps_image_embedding

RUN rm -rf /tmp/cellmaps_image_embedding

RUN cd /opt ;git clone https://github.com/CellProfiling/densenet.git
RUN cd /opt/densenet; git checkout 19dad140b4ba4a279f05bd7f91084911da7ade04
RUN pip install -r /opt/densenet/predict/requirements.txt

# at what point do we just make our own repo and own this?
COPY docker/predict_d121.py /opt/densenet/predict
COPY docker/dataset.py /opt/densenet/predict

RUN mkdir /opt/densenet/models
RUN wget https://github.com/CellProfiling/densenet/releases/download/v0.1.0/external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final.pth -O /opt/densenet/models/model.pth

ENTRYPOINT ["/opt/conda/bin/cellmaps_image_embeddingcmd.py"]

CMD ["--help"]
