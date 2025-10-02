pipeline {
    agent any

    environment{
        VENV_DIR = 'venv'
        GCP_PROJECT = 'mlops-2-472323'
        GCLOUD_PATH = '/var/jenkins_home/google-cloud-sdk/bin'
        KUBECTL_AUTH_PLUGIN = '/usr/lib/google-cloud-sdk/bin'
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
                        
                        echo "Checking for model weights..."
                        ls artifacts/models/
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
                
                # Verify weights exist in Jenkins workspace
                echo "=== Checking weights in Jenkins workspace ==="
                ls -lah artifacts/models/
                
                gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                gcloud config set project ${GCP_PROJECT}
                gcloud auth configure-docker --quiet
                docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .
                
                # Check inside the built image
                echo "=== Checking weights inside Docker image ==="
                docker run --rm gcr.io/${GCP_PROJECT}/ml-project:latest ls -lah artifacts/models/ || echo "Directory not found in image"
                docker run --rm gcr.io/${GCP_PROJECT}/ml-project:latest sh -c "test -f artifacts/models/best_recommender_model.weights.h5 && echo '✓ Weights exist in image' || echo '✗ Weights MISSING in image'"
                
                docker push gcr.io/${GCP_PROJECT}/ml-project:latest
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
                        gcloud container clusters get-credentials ml-app-cluster --region us-west2
                        kubectl apply -f deployment.yaml
                        '''
                    }
                }
            }
        }
        
    }
}