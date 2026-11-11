# Index artifacts bucket and a read-only policy for the search service.
# Applied against LocalStack only (see docs/infra-status.md); nothing here has been created in a real AWS account.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.70" }
  }
}

variable "endpoint" {
  description = "AWS API endpoint. Defaults to a local LocalStack."
  type        = string
  default     = "http://localhost:4566"
}

variable "bucket_name" {
  type    = string
  default = "wikiret-index-artifacts"
}

provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true
  endpoints {
    s3  = var.endpoint
    iam = var.endpoint
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket = var.bucket_name
}

# Versioning keeps every index build, so a bad build can be rolled back by pointing at the previous object version.
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_iam_policy_document" "read_artifacts" {
  statement {
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }
}

resource "aws_iam_policy" "read_artifacts" {
  name   = "wikiret-read-index-artifacts"
  policy = data.aws_iam_policy_document.read_artifacts.json
}

output "bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "read_policy_arn" {
  value = aws_iam_policy.read_artifacts.arn
}
