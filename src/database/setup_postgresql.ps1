# Setup PostgreSQL for Dunnhumby The Complete Journey dataset
Write-Host "Setting up PostgreSQL for Dunnhumby The Complete Journey dataset..." -ForegroundColor Green

# Set variables
$downloadUrl = "https://get.enterprisedb.com/postgresql/postgresql-14.5-1-windows-x64.exe"
$installerPath = "$env:TEMP\postgresql_installer.exe"
$pgPassword = "postgres" # Set your desired password here
$pgBinPath = "C:\Program Files\PostgreSQL\14\bin"
$workspacePath = "C:\Users\alexh\Desktop\The Complete Journey"
$schemaFile = "$workspacePath\create_schema.sql"
$importFile = "$workspacePath\import_data.sql"

# Download PostgreSQL installer
Write-Host "Downloading PostgreSQL installer..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath
    Write-Host "PostgreSQL installer downloaded successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to download PostgreSQL installer. Error: $_" -ForegroundColor Red
    exit 1
}

# Install PostgreSQL silently
Write-Host "Installing PostgreSQL..." -ForegroundColor Yellow
try {
    Start-Process -FilePath $installerPath -ArgumentList "--mode unattended --superpassword $pgPassword" -Wait
    Write-Host "PostgreSQL installed successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to install PostgreSQL. Error: $_" -ForegroundColor Red
    exit 1
}

# Add PostgreSQL bin directory to PATH temporarily
$env:Path += ";$pgBinPath"

# Wait for PostgreSQL service to start
Write-Host "Waiting for PostgreSQL service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Create the database and schema
Write-Host "Creating database schema..." -ForegroundColor Yellow
try {
    # Run the schema creation script
    $env:PGPASSWORD = $pgPassword
    & "$pgBinPath\psql" -U postgres -f $schemaFile
    Write-Host "Database schema created successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to create database schema. Error: $_" -ForegroundColor Red
    exit 1
}

# Import data
Write-Host "Importing data..." -ForegroundColor Yellow
try {
    # Run the data import script
    & "$pgBinPath\psql" -U postgres -f $importFile
    Write-Host "Data imported successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to import data. Error: $_" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "Verifying database setup..." -ForegroundColor Yellow
try {
    $query = "SELECT COUNT(*) FROM household_demographics;"
    $result = & "$pgBinPath\psql" -U postgres -d dunnhumby -c $query -t
    Write-Host "Verification successful. Number of household records: $result" -ForegroundColor Green
} catch {
    Write-Host "Failed to verify database setup. Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host "PostgreSQL setup complete!" -ForegroundColor Green
Write-Host "You can now connect to the database with the following credentials:" -ForegroundColor Green
Write-Host "Host: localhost" -ForegroundColor Green
Write-Host "Port: 5432" -ForegroundColor Green
Write-Host "Database: dunnhumby" -ForegroundColor Green
Write-Host "Username: postgres" -ForegroundColor Green
Write-Host "Password: $pgPassword" -ForegroundColor Green 