FROM python:3.7

LABEL maintainer="peter.haro@sintef.no"

RUN apt update && apt install python-dev -y
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-"]
CMD ["app:app"]