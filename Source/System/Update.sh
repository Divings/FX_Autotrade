#!/usr/bin/bash

yum makecache
systemctl stop fx-autotrade.service
yum update fx_autotrade-system -y
systemctl restart fx-autotrade.service