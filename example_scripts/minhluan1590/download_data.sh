export DATA_DIR=./atlas_data
export SIZE=xl

# Make sure the working directory is the root of the repository by checking that the following file exists: ./train.py
if [ ! -f ./train.py ]; then
    echo "Please make sure the working directory is the root of the repository"
    exit 1
fi

python preprocessing/prepare_qa.py --output_directory ${DATA_DIR} && \
python preprocessing/download_corpus.py --corpus corpora/wiki/enwiki-dec2018 --output_directory ${DATA_DIR} && \
python preprocessing/download_model.py --model models/atlas/${SIZE} --output_directory ${DATA_DIR} && \
python preprocessing/download_index.py --index indices/atlas/wiki/${SIZE} --output_directory ${DATA_DIR}
