.PHONY: install run test lint eval docker

install:
	pip install -r backend/requirements-dev.txt

run:
	cd backend && python app.py

test:
	pytest -q

lint:
	ruff check backend

eval:
	cd backend && python evaluate.py

docker:
	docker build -t fraudsentinel . && docker run --rm -p 8000:8000 fraudsentinel
