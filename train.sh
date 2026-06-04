# you can change densify_grad_abs_threshold and percent_dense for complex scenes
declare -a scenes=("./data/bike")
declare -a models=("./output/bike")
# Iterate over each scene and run training and rendering commands
for i in "${!scenes[@]}"; do
  echo "Processing scene ${i+1}: ${scenes[i]}"

  python train.py -s "${scenes[i]}" -m "${models[i]}" --eval
  python render.py -m "${models[i]}"

  echo "Scene ${i+1} processed successfully."
done