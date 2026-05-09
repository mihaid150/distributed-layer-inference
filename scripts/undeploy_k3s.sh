#!/usr/bin/env bash
set -e

kubectl delete namespace inference --ignore-not-found=true
