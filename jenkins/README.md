# Jenkins CI/CD Setup for PSGAN

## Prerequisites
- Jenkins server with Docker support
- Docker Hub or registry credentials
- GitHub/GitLab integration

## Setup Steps

1. **Install Jenkins Plugins**
   - Docker Pipeline
   - GitHub Integration
   - Blue Ocean (optional)

2. **Configure Credentials**
   - Add Docker registry credentials
   - Setup GitHub webhook

3. **Create Pipeline Job**
   - New Item → Pipeline
   - Pipeline script from SCM
   - Repository URL and branch
   - Script Path: `jenkins/Jenkinsfile`

4. **Environment Variables**
   - `DOCKER_CREDENTIALS_ID`: Docker registry credentials
   - `REGISTRY_URL`: Your Docker registry URL

## Pipeline Stages
1. **Checkout**: Get source code
2. **Build**: Create Docker image
3. **Test**: Run health checks
4. **Security Scan**: Vulnerability scanning
5. **Push**: Upload to registry
6. **Deploy**: Deploy to environment

## Webhooks
Setup GitHub webhook to trigger builds:
- Payload URL: `http://your-jenkins/github-webhook/`
- Content type: `application/json`
- Events: Push, Pull requests
