FROM python:3.7-slim


RUN apt update \
  && apt install -y libgl1-mesa-glx
  
COPY ./requirements.txt /home
RUN pip install -r /home/requirements.txt

WORKDIR /home

ENTRYPOINT ["python"]
COPY example2.png /home/example2.png
COPY test.py  /home/test.py

CMD ["test.py"]