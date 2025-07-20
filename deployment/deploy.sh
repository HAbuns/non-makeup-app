#!/bin/bash

# PSGAN Makeup Transfer App - Deployment Script

set -e

echo "🎨 PSGAN Makeup Transfer Deployment Script"
echo "=========================================="

# Configuration
APP_NAME="psgan-makeup-app"
IMAGE_NAME="psgan-webapp"
CONTAINER_NAME="psgan-container"
PORT="5000"

# Functions
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    echo "✅ Docker is available"
}

build_image() {
    echo "🔨 Building Docker image..."
    docker build -t $IMAGE_NAME:latest .
    echo "✅ Docker image built successfully"
}

stop_existing() {
    echo "🛑 Stopping existing container if running..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo "✅ Cleaned up existing container"
}

run_container() {
    echo "🚀 Starting new container..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:5000 \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/datasets/results:/app/datasets/results \
        --restart unless-stopped \
        $IMAGE_NAME:latest
    
    echo "✅ Container started successfully"
    echo "🌐 Application available at: http://localhost:$PORT"
}

run_with_compose() {
    echo "🐳 Using Docker Compose for deployment..."
    docker-compose down 2>/dev/null || true
    docker-compose up -d --build
    echo "✅ Docker Compose deployment completed"
    echo "🌐 Application available at: http://localhost:80"
}

show_status() {
    echo ""
    echo "📊 Container Status:"
    docker ps | grep $APP_NAME || docker ps | grep psgan
    echo ""
    echo "📝 Recent logs:"
    docker logs --tail 20 $CONTAINER_NAME 2>/dev/null || docker-compose logs --tail 20 2>/dev/null || true
}

# Main deployment logic
main() {
    case "${1:-docker}" in
        "docker")
            check_docker
            build_image
            stop_existing
            run_container
            show_status
            ;;
        "compose")
            check_docker
            run_with_compose
            show_status
            ;;
        "build")
            check_docker
            build_image
            ;;
        "stop")
            stop_existing
            docker-compose down 2>/dev/null || true
            echo "✅ All containers stopped"
            ;;
        "logs")
            echo "📝 Application logs:"
            docker logs -f $CONTAINER_NAME 2>/dev/null || docker-compose logs -f 2>/dev/null || echo "No containers running"
            ;;
        "status")
            show_status
            ;;
        *)
            echo "Usage: $0 [docker|compose|build|stop|logs|status]"
            echo ""
            echo "Commands:"
            echo "  docker  - Build and run with Docker (default)"
            echo "  compose - Build and run with Docker Compose"
            echo "  build   - Only build the Docker image"
            echo "  stop    - Stop all running containers"
            echo "  logs    - Show application logs"
            echo "  status  - Show container status"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
