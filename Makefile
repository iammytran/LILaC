# Khai báo các target không tạo ra file thực tế (chỉ là tên câu lệnh)
.PHONY: all embed retrieve visualize_result run_lilac

# Lệnh mặc định khi gõ "make" không truyền tham số
all: run_lilac

embed: 
	@echo "=== [1/3] Embedding InfoVQA ==="
	@bash -c 'eval "$$(conda shell.bash hook)" && conda activate mmembed && ./scripts/parse_multimodal_document/step8_embed_mmembed.sh -b "InfoVQA"'

retrieve: embed
	@echo "=== [2/3] Retrieving ==="
	@bash -c 'eval "$$(conda shell.bash hook)" && conda activate lilac-mmembed && CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/experiments/retriever_accuracy.sh -e "MM-Embed" -b "InfoVQA"'

visualize_result: embed retrieve
	@echo "=== [3/3] Visualizing result ==="
	./scripts/experiment_visualization/retriever_accuracy.sh 

run_lilac: visualize_result
	@echo "✅ ALL STEPS COMPLETED SUCCESSFULLY!"
