#!/usr/bin/env python3
"""Check commit-message and merge-request-description lengths."""

import argparse
import json
import os
import sys
import urllib.request


def gitlab_api(path):
  url     = '%s/projects/%s/merge_requests/%s%s' % (os.environ['CI_API_V4_URL'], os.environ['CI_PROJECT_ID'], os.environ['CI_MERGE_REQUEST_IID'], path)
  request = urllib.request.Request(url)
  request.add_header('JOB-TOKEN', os.environ['CI_JOB_TOKEN'])
  with urllib.request.urlopen(request, timeout=30) as response:
    return json.load(response), response.headers.get('X-Next-Page')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--commit-message-limit', type=int, required=True)
  parser.add_argument('--mr-description-limit', type=int, required=True)
  args = parser.parse_args()

  merge_request, _ = gitlab_api('')
  errors           = []
  description      = merge_request.get('description') or ''
  if len(description) > args.mr_description_limit:
    errors.append('Merge request description is too long: %d characters (maximum %d).' % (len(description), args.mr_description_limit))

  page = '1'
  while page:
    commits, page = gitlab_api('/commits?per_page=100&page=%s' % page)
    for commit in commits:
      message = commit['message'].rstrip('\n')
      if len(message) > args.commit_message_limit:
        errors.append('Commit %s is too long: %d characters (maximum %d).' % (commit['id'][:12], len(message), args.commit_message_limit))

  if errors:
    sys.exit('\n'.join(errors))


if __name__ == '__main__':
  sys.exit(main())
