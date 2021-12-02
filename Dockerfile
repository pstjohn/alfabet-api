FROM continuumio/miniconda3

WORKDIR /api
RUN conda install -c conda-forge mamba
COPY environment.yml .
RUN mamba env update -f environment.yml && \
    conda clean --all --yes
COPY api.py .
COPY tests .

# ENV PYTHONPATH "${PYTHONPATH}:/deploy/app"

EXPOSE 8000

#ENTRYPOINT "/bin/bash"
#CMD gunicorn --bind 0.0.0.0:$PORT wsgi:app
CMD ["uvicorn", "api:api", "--host=0.0.0.0"]
