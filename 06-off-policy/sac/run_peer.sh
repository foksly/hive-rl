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

python3 sac.py \
  --peer-id ${PEER_ID} \
  --batch-size 256 \
  --target-batch-size 10000 \
  --lr 0.0001 \
  --use-local-updates \
