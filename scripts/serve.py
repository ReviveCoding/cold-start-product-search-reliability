from _bootstrap import bootstrap_src

bootstrap_src()

import uvicorn


if __name__ == "__main__":
    uvicorn.run("product_search.serving.app:app", host="0.0.0.0", port=8000, reload=False)
