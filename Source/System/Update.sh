#!/usr/bin/bash

yum makecache
systemctl stop fx-autotrade.service
yum update fx_autotrade-system
systemctl restart fx-autotrade.service