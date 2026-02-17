import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const BUCKET = "anuc-satellite-analysis";

let _s3: S3Client | null = null;

function getS3(): S3Client {
  if (!_s3) {
    _s3 = new S3Client({
      region: process.env.AWS_DEFAULT_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
      },
    });
  }
  return _s3;
}

export async function getPresignedUrl(s3Key: string): Promise<string> {
  const command = new GetObjectCommand({
    Bucket: BUCKET,
    Key: s3Key,
  });
  return getSignedUrl(getS3(), command, { expiresIn: 3600 });
}
