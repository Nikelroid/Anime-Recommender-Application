pipeline {
    agent any

    stages{
        stage('Clone from github'){
            steps{
                script{
                    echo 'Cloning from github ...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'jenkins-github-access', url: 'https://github.com/Nikelroid/Anime-Recommender-Application']])
                }
            }
        }
    }
}