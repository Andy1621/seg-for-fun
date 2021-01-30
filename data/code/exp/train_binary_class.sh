echo "=====Start training binary-class models====="

if [ $1 == true ]; then
    bash ./exp/train_binary_class_all.sh
fi

echo "=====Training binary-class models over====="