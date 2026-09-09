FROM python:3.11-slim

WORKDIR /app

# System deps in case any wheel needs to compile from source on the target platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir jupyterlab ipykernel

COPY . .

EXPOSE 8888

# Default: interactive Jupyter Lab. The notebooks read pre-computed results from
# results/ (already in the repo), so they run immediately without re-training anything.
#
# To instead regenerate everything from scratch (re-run every experiment, which
# retrains PINNs and can take a while):
#   docker run <image> bash -c "python experiments/dimensional_scaling.py && \
#     python experiments/convergence.py && python experiments/parameterized_solver.py && \
#     python experiments/inverse_problem.py"
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
