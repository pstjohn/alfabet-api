FROM continuumio/miniconda3

WORKDIR /api
RUN conda install -c conda-forge mamba
COPY environment.yml .
RUN mamba env update -f environment.yml && \
    conda clean --all --yes
COPY api.py .
COPY tests .

EXPOSE 8000

CMD ["uvicorn", "api:api", "--host=0.0.0.0"]
