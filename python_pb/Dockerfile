FROM python:3.12-slim

COPY ./requirements.txt /home
RUN pip install --no-cache-dir -r /home/requirements.txt

WORKDIR /home

ENTRYPOINT ["python"]
COPY example2.png /home/example2.png
COPY test.py  /home/test.py

CMD ["test.py"]