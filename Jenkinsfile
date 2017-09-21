pipeline {
    agent any

    stages {
        stage('Warnings - cpplint') {
            steps {
                sh "make cpplint || echo '--- cpplint has errors ---'"
                warnings consoleParsers: [[parserName: 'CppLint']], failedTotalAll: '0', usePreviousBuildAsReference: true
            }
        }
        stage('Dependencies') {
            steps {
                sh 'make test-math-dependencies'
            }
        }
        stage('Doxygen') {
            steps {
                sh 'make clean-all'
                sh 'make doxygen'
            }
        }
        stage('Headers') {
            steps {
                sh "make -j${env.PARALLEL} test-headers"
            }
        }
        //stage('Unit') {
        //    steps {
        //        sh "./runTests.py -j${env.PARALLEL} test/unit"
        //    }
        //}
        //stage('Distribution tests') {
        //    steps {
        //        sh './runTests.py -j${env.PARALLEL} test/prob'
        //    }
        //}
    }
}
