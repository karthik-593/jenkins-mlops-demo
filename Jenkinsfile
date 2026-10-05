pipeline {
  agent any

  parameters {
    string(name: 'THRESHOLD', defaultValue: '0.85',
           description: 'Minimum accuracy required to pass the quality gate')
    string(name: 'CLASS_SEP', defaultValue: '1.0',
           description: 'Data difficulty (lower = harder = lower accuracy)')
  }

  options {
    timestamps()
    disableConcurrentBuilds()
  }


  environment {
    VENV = "${WORKSPACE}/.venv"
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Setup') {
      steps {
        sh '''
          python3 -m venv "$VENV"
          . "$VENV/bin/activate"
          pip install --upgrade pip
          pip install -r requirements.txt
        '''
      }
    }

    stage('Generate Data') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python3 src/generate_data.py --class-sep ${CLASS_SEP}
        '''
      }
    }

    stage('Validate Data') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python3 src/validate_data.py
        '''
      }
    }

    stage('Train') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python3 src/train.py
        '''
      }
    }

    stage('Evaluate & Quality Gate') {
      steps {
        sh '''
          . "$VENV/bin/activate"
          python3 src/evaluate.py --threshold ${THRESHOLD} --metric accuracy
        '''
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'artifacts/*, data/*.csv',
                       allowEmptyArchive: true, fingerprint: true
    }
    success { echo 'Build OK — model passed the quality gate.' }
    failure { echo 'Build FAILED — check which stage broke (validation / gate / setup).' }
    cleanup { cleanWs() }
  }
}
