.PHONY: setup setup-ci run-ui run-api test smoke compile clean docker-up

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

setup-ci:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements-ci.txt

run-ui:
	streamlit run app/ui/product_app.py

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

compile:
	python -m compileall .

smoke:
	python scripts/smoke_run.py

docker-up:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
