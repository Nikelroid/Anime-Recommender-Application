pipeline {
    agent any

    environment{
        VENV_DIR = 'venv'

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
                        '''
                    }
                }
            }
        }
    }
}