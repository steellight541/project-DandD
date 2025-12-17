FROM python:3.10.16

WORKDIR /

RUN pip install \
    uvicorn \
    fastapi \
    gradio \
    scikit-learn


COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./DandD /DandD
COPY ./src/DandD_app .

CMD ["python", "page.py"]