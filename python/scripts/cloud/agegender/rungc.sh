gcloud ml-engine jobs submit training agegender_$(date +%Y%m%d_%H%M%S) --job-dir gs://subramgo/agegender --module-name trainer.agegender --package-path ./trainer
