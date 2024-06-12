# create a build env w/ rocm dylibs & c compiler
from rocm/dev-ubuntu-22.04:6.0-complete as build

# install rust + random deps
run apt update \
 && apt upgrade -y \
 && apt install -y pkg-config libssl-dev cmake libclang-dev libomp-dev

run curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y

env PATH="/root/.cargo/bin:${PATH}"

# build demesne
workdir /app

copy . .

env RUSTFLAGS="-C target-cpu=native"

run --mount=type=cache,target=/target cargo build --release --features rocm --target-dir /target \
 && cp /target/release/demesne /bin/demesne

# minimize as much as we can
from rocm/dev-ubuntu-22.04:6.0-complete
copy --from=build /bin/demesne /bin/demesne
