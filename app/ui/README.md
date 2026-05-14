# UI surfaces

`product_app.py` is the modern command-center UI and should be used for demos.

`v06_app.py` is kept as the canonical backwards-compatible entry point and delegates to `product_app.py`.

Run either:

```bash
streamlit run app/ui/product_app.py
# or
streamlit run app/ui/v06_app.py
```
