FROM Python 3.13.3
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 50000
CMD ['python','app.py']