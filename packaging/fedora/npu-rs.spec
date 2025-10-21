
Name:           npu-rs
Version:        1.0.0
Release:        1%{?dist}
Summary:        Rust NPU driver for RISC boards
License:        GPL-3.0-or-later
URL:            https://github.com/KushalMeghani1644/NPU-rs
Source0:        %{name}-%{version}.tar.gz
AutoReqProv: no
BuildRequires:  cargo, rust, rpm-build

%description
Rust NPU driver for RISC boards.

# Prepare source
%prep
# Unpack into BUILD/npu-rs-1.0.0
%setup -q -n npu-rs-1.0.0

# Build with cargo
%build
cargo build --release --locked --manifest-path=%{_builddir}/npu-rs-1.0.0/Cargo.toml

# Install the binary
%install
mkdir -p %{buildroot}/usr/bin
cp %{_builddir}/npu-rs-1.0.0/target/release/npu-rs %{buildroot}/usr/bin/

# Files to include in RPM
%files
/usr/bin/npu-rs
%license LICENSE
%doc README.md

# Changelog
%changelog
* Tue Oct 21 2025 KushalMeghani1644 <kushalmeghani108@gmail.com> - 1.0.0-1
- Initial package
