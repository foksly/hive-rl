while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--peer-id)
      PEER_ID="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
  esac
done

python3 ppo_hivemind.py \
  --peer-id ${PEER_ID} \
  --batch-size 512 \
  --target-batch-size 100000 \
  --n-envs 8 \
  --n-rollout-steps 128 \
  --lr 0.001 \
  --experiment-prefix 8_peers\
