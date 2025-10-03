pipeline {
    agent any

    environment{
        VENV_DIR = 'YOUR_VENV_DIR'
        GCP_PROJECT = 'YOUR_GCP_PROJECT_ID'
        GCLOUD_PATH = 'YOUR_GCLOUD_PATH'
        KUBECTL_AUTH_PLUGIN = 'YOUR_KUBERNETES_AUTH'
    }

    stages{
        stage('Clone from github'){
            steps{
                script{
                    echo 'Cloning from github ...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'jenkins-github-access', url: 'https://github.com/Nikelroid/Anime-Recommender-Application']])
                }
            }
        }
        
        stage('Make Virtual Environment'){
            steps{
                script{
                    echo 'Making New Virtual Environment ...'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    pip install dvc
                    '''
                }
            }
        }
        
        stage('DVC Pull'){
            steps{
                withCredentials([file(credentialsId:'gcp-key',variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'DVC Pull started ...'
                        sh '''
                        . ${VENV_DIR}/bin/activate
                        dvc pull
                        echo "Verifying model files:"
                        ls -lah artifacts/models/
                        '''
                    }
                }
            }
        }
        
        stage('Build and push Image to gcr'){
            steps{
                withCredentials([file(credentialsId:'gcp-key',variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Building and pushing Image to gcr ...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet
                        
                        # Build with explicit platform
                        docker build --platform linux/amd64 \
                            -t gcr.io/${GCP_PROJECT}/ml-project:latest .
                        
                        # Push to GCR
                        docker push gcr.io/${GCP_PROJECT}/ml-project:latest
                        
                        echo "=== Image pushed successfully ==="
                        '''
                    }
                }
            }
        }
        
        stage('Verify Image in GCR'){
            steps{
                withCredentials([file(credentialsId:'gcp-key',variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Verifying image in GCR ...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        
                        echo "=== Image Details ==="
                        gcloud container images describe gcr.io/${GCP_PROJECT}/ml-project:latest --format=json
                        '''
                    }
                }
            }
        }
        
        stage('Setup Image Pull Secret'){
            steps{
                withCredentials([file(credentialsId:'gcp-key',variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Creating image pull secret ...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}:${KUBECTL_AUTH_PLUGIN}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud container clusters get-credentials ml-app-cluster --region us-west2
                        
                        # Delete old secret if exists
                        kubectl delete secret gcr-json-key --ignore-not-found
                        
                        # Create new secret
                        kubectl create secret docker-registry gcr-json-key \
                            --docker-server=gcr.io \
                            --docker-username=_json_key \
                            --docker-password="$(cat ${GOOGLE_APPLICATION_CREDENTIALS})" \
                            --docker-email=jenkins@mlops.com
                        
                        echo "=== Secret created ==="
                        kubectl get secret gcr-json-key
                        '''
                    }
                }
            }
        }

        stage('Deploy to Kubernetes'){
            steps{
                withCredentials([file(credentialsId:'gcp-key',variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Deploying to Kubernetes ...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}:${KUBECTL_AUTH_PLUGIN}
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud container clusters get-credentials YOUR_KUBERNETER_CLUSTER_NAME --region YOUR_REGION
                        
                        kubectl apply -f deployment.yaml
                        
                        echo "=== Waiting for deployment ==="
                        kubectl rollout status deployment/ml-app --timeout=300s
                        
                        echo "=== Pod Status ==="
                        kubectl get pods -l app=ml-app
                        
                        echo "=== Pod Events ==="
                        kubectl get events --sort-by=.metadata.creationTimestamp | tail -20
                        '''
                    }
                }
            }
        }
    }
}