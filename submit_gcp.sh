GCS_BUCKET="gs://duke-bme590-cz/ds-cyclegan/ai_platform/experiments"
IMAGE_URI=gcr.io/my-project-1475521763853/dscycle-gan:tf2
REGION=us-east1

docker build -f Dockerfile -t $IMAGE_URI .
docker push $IMAGE_URI

JOB_NAME=DScyclegan_baseGAN$(date +%Y_%m_%d_%H%M%S)
JOB_NAME=monet2photo_5idloss_noskip$(date +%Y_%m_%d_%H%M%S)
JOB_DIR=$GCS_BUCKET"/"$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
  --master-image-uri $IMAGE_URI \
  --scale-tier custom \
  --master-machine-type standard_p100 \
  --region $REGION \
  --job-dir $JOB_DIR \
  -- \
  --bs 1 \
  --in_h 256 \
  --in_w 256 \
  --epochs 100 \
  --disc_loss 1 \
  --id_loss 5 \
  --discriminator_norm instance \
  --generator_norm instance
gcloud ai-platform jobs describe $JOB_NAME
