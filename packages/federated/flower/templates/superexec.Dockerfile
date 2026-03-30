FROM flwr/superexec:1.27.0

WORKDIR /app

COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]
