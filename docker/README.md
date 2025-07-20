# Docker Commands for PSGAN Makeup App

## Development
```bash
# Build and run development
docker-compose -f docker/docker-compose.yml up --build

# Run in background
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

## Production
```bash
# Build and run production
docker-compose -f docker/docker-compose.prod.yml up --build

# Deploy with script
./deployment/deploy.sh production
```

## Useful Commands
```bash
# Stop all containers
docker-compose down

# Remove all containers and volumes
docker-compose down -v

# Check health
curl http://localhost:5000/health

# View container stats
docker stats psgan-makeup-app
```
