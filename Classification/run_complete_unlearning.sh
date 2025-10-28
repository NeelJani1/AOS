#!/bin/bash

BASE_DIR="results/adaptive_otsu_fixed_all_20251026_022500"
MODEL_PATH="/home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/Classification/saved_models_regularized_new/0model_SA_best.pth.tar"

declare -A methods
methods[GA]=0.001
methods[FT]=0.1
methods[RL]=0.1

# All percentages from 10% to 100%
percentages=(10 20 30 40 50 60 70 80 90 100)

echo "🚀 Starting COMPLETE unlearning experiments (10% to 100%)..."
echo "📊 Methods: GA, FT, RL"
echo "🎯 Percentages: ${percentages[*]}%"
echo "=========================================="

for method in "${!methods[@]}"; do
    echo ""
    echo "🎯 PROCESSING METHOD: $method"
    echo "=========================================="
    
    for percent in "${percentages[@]}"; do
        echo ""
        echo "📊 Running $method with ${percent}% forgetting..."
        
        MASK_PATH="$BASE_DIR/masks/$method/${percent}percent/mask_otsu_${method}_${percent}percent_conservative.pt"
        SAVE_DIR="$BASE_DIR/unlearning/$method/${percent}percent"
        LR=${methods[$method]}
        
        # Check if mask exists (handles FT/RL 100% skip automatically)
        if [ ! -f "$MASK_PATH" ]; then
            echo "⏭️  Skipping $method ${percent}% - mask not found (expected for FT/RL at 100%)"
            continue
        fi
        
        # Create save directory
        mkdir -p "$SAVE_DIR"
        
        # Set forget percentage (handle 100% case)
        if [ "$percent" -eq 100 ]; then
            FORGET_PERCENT="1.0"
        else
            FORGET_PERCENT="0.$percent"
        fi
        
        echo "🔧 Configuration:"
        echo "   Method: $method"
        echo "   Forget: ${percent}% ($FORGET_PERCENT)"
        echo "   Learning Rate: $LR"
        echo "   Mask: $MASK_PATH"
        echo "   Save: $SAVE_DIR"
        
        # Run unlearning
        python main_forget.py \
            --unlearn $method \
            --forget_percentage $FORGET_PERCENT \
            --mask_path "$MASK_PATH" \
            --save_dir "$SAVE_DIR" \
            --dataset cifar100 \
            --model_path "$MODEL_PATH" \
            --batch_size 128 \
            --unlearn_epochs 10 \
            --unlearn_lr $LR \
            --seed 2
            
        if [ $? -eq 0 ]; then
            echo "✅ SUCCESS: $method ${percent}% completed"
        else
            echo "❌ FAILED: $method ${percent}% - check logs"
        fi
        
        echo "----------------------------------------"
    done
    echo "✅ COMPLETED METHOD: $method"
done

echo ""
echo "🎉 ALL UNLEARNING EXPERIMENTS COMPLETED!"
echo "📁 Results saved in: $BASE_DIR/unlearning/"
echo "📊 Ready for analysis!"
