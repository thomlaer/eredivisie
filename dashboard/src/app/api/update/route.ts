import crypto from "node:crypto";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const REPOSITORY = "thomlaer/eredivisie";
const WORKFLOW = "rebuild-predictions.yml";

function safeEquals(left: string, right: string): boolean {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && crypto.timingSafeEqual(leftBuffer, rightBuffer);
}

export async function POST(request: Request) {
  const expectedCode = process.env.UPDATE_CODE || "";
  const token = process.env.GITHUB_ACTIONS_TOKEN || "";
  if (!expectedCode || !token) {
    return NextResponse.json({ ok: false, message: "De updateknop moet nog eenmalig worden ingesteld." }, { status: 503 });
  }

  let submittedCode = "";
  try {
    const body = (await request.json()) as { code?: unknown };
    submittedCode = typeof body.code === "string" ? body.code.trim() : "";
  } catch {
    return NextResponse.json({ ok: false, message: "Ongeldige aanvraag." }, { status: 400 });
  }
  if (!submittedCode || !safeEquals(submittedCode, expectedCode)) {
    return NextResponse.json({ ok: false, message: "De code klopt niet." }, { status: 401 });
  }

  const response = await fetch(`https://api.github.com/repos/${REPOSITORY}/actions/workflows/${WORKFLOW}/dispatches`, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: JSON.stringify({ ref: "main" }),
  });
  if (!response.ok) {
    return NextResponse.json({ ok: false, message: "GitHub kon de update niet starten." }, { status: 502 });
  }
  return NextResponse.json({
    ok: true,
    message: "Bijwerken is gestart. De website vernieuwt automatisch wanneer GitHub klaar is.",
    actionsUrl: `https://github.com/${REPOSITORY}/actions/workflows/${WORKFLOW}`,
  });
}
