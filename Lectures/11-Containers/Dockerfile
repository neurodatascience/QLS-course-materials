FROM python:3.9-slim

# Create a user group and a user so we don't run this as root
RUN groupadd --gid 1000 qls_user \    
    && useradd --uid 1000 --gid 1000 -m qls_user
# Set the user to our newly created user
USER qls_user

RUN pip install numpy nilearn scikit-learn matplotlib seaborn pandas

WORKDIR /predict
COPY ./ABIDE_classification.py /predict

CMD ["/bin/bash"]